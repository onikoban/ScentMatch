def generate_response(query, results):

    response = (
        f"\nHere are some recommendations for:\n"
        f"'{query}'\n"
    )

    for perfume in results:

        response += (
            f"\n━━━━━━━━━━━━━━━━━━\n"
            f"{perfume['name']}\n\n"
            f"Notes: {perfume['notes']}\n\n"
            f"{perfume['description'][:250]}...\n\n"
            f"Semantic Match Score: {perfume['ux_score']}%\n"
        )

    return response